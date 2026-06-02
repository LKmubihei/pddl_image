(define (problem picture_15)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery green_sensor blue_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_battery)
        (clear blue_battery)
        (clear green_sensor)
        (part_at blue_battery table)
        (part_at green_sensor table)
        (part_at red_battery table)
    )
    
    (:goal
        (and
            (part_at red_battery battery_placement)
        )
    )
)
