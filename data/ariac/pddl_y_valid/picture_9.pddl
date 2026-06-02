(define (problem picture_9)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery blue_sensor red_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear red_regulator)
        (on red_regulator blue_sensor)
        (part_at blue_sensor table)
        (part_at blue_battery table)
    )
    
    (:goal
        (and
            (part_at red_regulator regulator_placement)
            (part_at blue_battery battery_placement)
        )
    )
)
