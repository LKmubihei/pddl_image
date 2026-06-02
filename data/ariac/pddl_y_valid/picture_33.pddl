(define (problem picture_33)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_sensor red_regulator blue_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_sensor)
        (clear red_regulator)
        (clear blue_pump)
        (part_at red_regulator table)
        (on red_sensor blue_pump)
        (part_at blue_pump table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
            (part_at red_regulator regulator_placement)
        )
    )
)
